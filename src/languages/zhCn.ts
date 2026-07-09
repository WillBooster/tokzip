import { zhCnModule } from '../generated/zhCn.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(zhCnModule);

export { zhCnModule };
