import { haskellModule } from '../generated/haskell.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(haskellModule);

export { haskellModule };
